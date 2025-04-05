"""This is both the search tool definition and a script to build the search index. Build index with modal run search.py (no deploy key required)."""

import asyncio
import multiprocessing as mp
import pickle
import warnings
from itertools import starmap
from pathlib import Path

from fastapi import APIRouter
from hot_plasmids import plasmid_dict
from modal import App, Image, Volume, enter, method
from models import SearchResult

CPU_COUNT = 48  # a big number to spread out indexing

image = Image.debian_slim().pip_install("tantivy", "biopython", "aiofiles")

app = App()
genbank_files = Volume.from_name("genbank-files")
index_vol = Volume.from_name("genbank-index")
router = APIRouter()


def parse_genbank(file_path: Path):
    from Bio import BiopythonWarning, SeqIO

    warnings.simplefilter("ignore", BiopythonWarning)

    result = []
    gb_id = int(file_path.stem)

    for record in SeqIO.parse(file_path, "genbank"):
        # Process the main GenBank record
        main_entry = SearchResult(
            title=record.annotations.get("title", ""),
            body=(
                f"Definition: {record.description}\n"
                f"Keywords: {', '.join(record.annotations.get('keywords', []))}\n"
                f"Organism: {record.annotations.get('organism', '')}"
            ),
            docid=gb_id,
            featureid=0,
            sequence=str(record.seq),
        )
        result.append(main_entry)

        # Process features
        for feature in record.features:
            if "label" not in feature.qualifiers:
                continue
            try:
                body_parts = []
                if "note" in feature.qualifiers:
                    body_parts.append(f"Note: {feature.qualifiers['note'][0]}")
                if "product" in feature.qualifiers:
                    body_parts.append(f"Product: {feature.qualifiers['product'][0]}")
                if "gene" in feature.qualifiers:
                    body_parts.append(f"Gene: {feature.qualifiers['gene'][0]}")

                feature_entry = SearchResult(
                    title=feature.qualifiers["label"][0],
                    body="\n".join(body_parts),
                    docid=gb_id,
                    featureid=len(result),
                    sequence=str(feature.location.extract(record.seq)),
                )
            except AttributeError:
                continue
            result.append(feature_entry)
        break  # we don't know what to do with multiple records
    return result


with image.imports():
    import tantivy

    def get_schema():
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=False)
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_integer_field("docid", stored=True)
        schema_builder.add_integer_field("featureid", stored=True)
        return schema_builder.build()

    def get_index():
        index_location = Path("/search/index")
        # make sure the index directory exists
        index_location.mkdir(parents=True, exist_ok=True)
        return tantivy.Index(get_schema(), index_location.as_posix())


@app.function(
    image=image,
    volumes={"/seqs": genbank_files, "/search": index_vol},
    cpu=CPU_COUNT,
    timeout=10000,  # a really big number to prevent timeout
)
def build_index():
    index = get_index()
    records: dict[int, list[SearchResult]] = {}
    writer = index.writer()
    # clear the index, since we are rebuilding it
    writer.delete_all_documents()

    count = 0
    batch_size = 2**10
    files = list(Path("/seqs").glob("*.gbk"))

    # break them into batches
    with mp.Pool(processes=CPU_COUNT) as pool:
        for i in range(len(files) // batch_size + 1):
            batch = files[i * batch_size : (i + 1) * batch_size]
            results = pool.map(parse_genbank, batch)
            for result in results:
                count += 1
                if count % 100 == 0:
                    print(f"Processing file {count}")
                for record in result:
                    doc = tantivy.Document(
                        title=record.title,
                        body=record.body,
                        docid=record.docid,
                        featureid=record.featureid,
                    )
                    writer.add_document(doc)
                    records.setdefault(record.docid, []).append(record)
    writer.commit()
    writer.wait_merging_threads()

    # write records to disk
    with open("/search/records.pkl", "wb") as f:
        pickle.dump(records, f)

    index.reload()


@app.cls(
    image=image,
    allow_concurrent_inputs=10,
    volumes={"/search": index_vol, "/seqs": genbank_files},
    cpu=0.25,
    keep_warm=1,  # I like fast search results, but this is gonna cost us some $
)
class PlasmidSearch:
    @enter()
    def setup(self):
        self.index = get_index()
        with open("/search/records.pkl", "rb") as f:
            self.records = pickle.load(f)  # noqa: S301
        self.index.reload()

        self.hot_plasmid_norm = {
            k.strip().casefold(): v for k, v in plasmid_dict.items()
        }

    @method()
    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        import aiofiles

        async def process_result_searcher(score, doc_address=None, doc_id=None):
            record = None
            if doc_address:
                doc = searcher.doc(doc_address)
                record = self.records[doc["docid"][0]][doc["featureid"][0]]
                doc_id = doc["docid"][0]
            elif doc_id and doc_id in self.records and self.records[doc_id]:
                record = self.records[doc_id][0]
            if not doc_id or not record:
                return None
            record.score = score
            async with aiofiles.open(f"/seqs/{doc_id}.gbk") as f:
                record.genbank = await f.read()
            return record

        # first try to match hot plasmids
        if query.strip().casefold() in self.hot_plasmid_norm:
            ids = self.hot_plasmid_norm[query.strip().casefold()]
            hits = [(1.0, None, int(i)) for i in ids]
        else:
            searcher = self.index.searcher()
            query = self.index.parse_query(query, ["title", "body"])
            search = searcher.search(query, limit)
            hits = search.hits

        results = await asyncio.gather(*list(starmap(process_result_searcher, hits)))
        return [r for r in results if r]


@router.get("/search")
async def search_endpoint(query: str) -> list[SearchResult]:
    """
    Perform a keyword search for plasmids.

    This endpoint searches for plasmids in the description, title, or genbank components using the provided query.

    ### Query Syntax
    The search query uses the Tantivy query syntax, which supports the following operators:
    - `+`: Includes a term (must be present)
    - `-`: Excludes a term (must not be present)
    - `AND`: Requires both terms to be present
    - `OR`: Requires at least one of the terms to be present

    ### Args:
    - `query` (str): The search query string using Tantivy syntax.

    ### Returns:
    - `list[SearchResult]`: A list of search results ordered by relevance.
    """
    plasmid_search = PlasmidSearch()
    return await plasmid_search.search.remote.aio(query)


@app.local_entrypoint()
def main() -> None:
    # rebuild the search index
    build_index.remote()
