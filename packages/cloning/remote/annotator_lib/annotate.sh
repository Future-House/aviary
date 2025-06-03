# Usage: ./annotate.sh <input.fasta> <job_name>
# Description: This script takes a linear fasta file of plasmid and output annotated genbank file

NDB_LIST="extra"
PROT_LIST="swissprot fpbase"

# Check args
if [ $# -ne 2 ]; then
    echo "Usage: ./annotate.sh <input.fasta> <job_name>"
    exit 1
fi

# Check if input file exists
if [ ! -f $1 ]; then
    echo "Error: $1 not found"
    exit 1
fi

mkdir -p $2

#$MMSEQS createdb $2/circular.fasta $2/query.db --dbtype 2
$MMSEQS createdb $1 $2/query.db --dbtype 2

for N in $NDB_LIST; do
    # BLASTN with no composition bias correction/masking (would reduce fidelity of query)
    # sensitivity 3 - a bit less sensitive, since we expect exact matches
    # wrap to account for circular input
    $MMSEQS search $2/query.db $N $2/${N}_results.db tmp --search-type 3 --comp-bias-corr 0 --mask 0  --wrapped-scoring 1 -s 3
    # these headers must match python post-processing code (in modal)
    $MMSEQS convertalis $2/query.db $N $2/${N}_results.db $2/${N}_results.tsv --format-output "target,evalue,qstart,qend,qcov,tcov,theader"
done


for P in $PROT_LIST; do
    # extracts orfs and matches all reading frames/directions (6-frame translation)
    $MMSEQS search $2/query.db $P $2/${P}_results.db tmp --search-type 2 --comp-bias-corr 0 --mask 0
    $MMSEQS convertalis $2/query.db $P $2/${P}_results.db $2/${P}_results.tsv --format-output "target,evalue,qstart,qend,qframe,qcov,tcov,theader"
done


# merge all results
cat $2/*_results.tsv > $2/all_results.tsv
