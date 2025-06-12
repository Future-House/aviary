$MMSEQS createdb extra.fasta $1/extra --dbtype 2
$MMSEQS createdb fpbase.fasta $1/fpbase --dbtype 1
$MMSEQS databases UniProtKB/Swiss-Prot $1/swissprot /tmp
### we cannot preserve header information - so would require a bunch of work..
##mmseqs concatdbs swissprot fpbase.db proteins.db --threads 1
#mmseqs databases SILVA silva /tmp
##mmseqs concatdbs silva extra.db nucleotides.db --threads 1
