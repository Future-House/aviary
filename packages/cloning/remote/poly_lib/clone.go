package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/bebop/poly/clone"
	"github.com/bebop/poly/io/fasta"
	"github.com/bebop/poly/io/rebase"
	"github.com/bebop/poly/transform"
)

type EnzymeOutput struct {
	Name            string `json:"name"`
	RegexpFor       string `json:"regexp_for"`
	RegexpRev       string `json:"regexp_rev"`
	Skip            int    `json:"skip"`
	OverheadLength  int    `json:"overhead_length"`
	RecognitionSite string `json:"recognition_site"`
}

type Output struct {
	Enzyme      EnzymeOutput `json:"enzyme"`
	FastaOutput string       `json:"fasta_output"`
}

func main() {
	// Define command-line flags
	rebaseFile := flag.String("rebase", "", "Path to the REBASE data file")
	fastaFile := flag.String("fasta", "", "Path to the input FASTA file")
	enzymeNames := flag.String("enzymes", "", "Comma-separated names of the enzymes to use for cutting")
	ligate := flag.Bool("ligate", false, "Ligate the fragments after cutting")

	// Parse command-line flags
	flag.Parse()

	// Check if required flags are provided
	if *rebaseFile == "" || *enzymeNames == "" || *fastaFile == "" {
		fmt.Println("Usage: ./rebase-cli -rebase <path_to_rebase_file> -enzymes <enzyme1,enzyme2,...> -fasta <path_to_fasta_file>")
		os.Exit(1)
	}

	// Load REBASE data
	enzymeMap, err := rebase.Read(*rebaseFile)
	if err != nil {
		log.Fatalf("Error reading REBASE file: %v", err)
	}

	// Parse enzyme names
	enzymeList := strings.Split(*enzymeNames, ",")

	// Convert the specified enzymes
	var enzymes []clone.Enzyme
	for _, enzymeName := range enzymeList {
		enzymeName = strings.TrimSpace(enzymeName)
		rebaseEnzyme, exists := enzymeMap[enzymeName]
		if !exists {
			log.Fatalf("Enzyme %s not found in REBASE data", enzymeName)
		}

		enzyme, err := convertRebaseToCloneEnzyme(rebaseEnzyme)
		if err != nil {
			log.Fatalf("Error converting enzyme %s: %v", enzymeName, err)
		}
		enzymes = append(enzymes, enzyme)
	}

	// Read the FASTA file
	parts, err := readFastaFile(*fastaFile)

	if err != nil {
		log.Fatalf("Error reading FASTA file: %v", err)
	}

	// Initialize a slice to hold all fragments
	var allFragments []clone.Fragment

	// We assume the order of the enzymes matches
	// what was given.
	for _, enzyme := range enzymes {
		for _, part := range parts {

			// Cut the sequence with the specified enzyme
			fragments := clone.CutWithEnzyme(part, false, enzyme)


			// Append fragments to the allFragments slice
			allFragments = append(allFragments, fragments...)
		}
	}

	// Ligate the fragments if specified
	var fastaOutput string
	if *ligate {
		openConstructs, _ := clone.CircularLigate(allFragments)
		// We ignore infinite loops
		fastaOutput = generateFastaOutputFromConstructs(openConstructs)
	} else {
		// Generate FASTA output from all fragments
		fastaOutput = generateFastaOutputFromFragments(allFragments)
	}
	fmt.Println(fastaOutput)
	// paranoia
	os.Stdout.Sync()
}

func readFastaFile(path string) ([]clone.Part, error) {
	fastas, err := fasta.Read(path)
	if err != nil {
		return nil, err
	}

	var sequences []clone.Part
	for _, f := range fastas {
		isCircular := strings.Contains(strings.ToLower(f.Name), "(circular)")

		sequences = append(sequences, clone.Part{
			Sequence: f.Sequence,
			Circular: isCircular,
		})
	}

	return sequences, nil
}

func generateFastaOutputFromConstructs(constructs []string) string {
	var fastaBuilder strings.Builder

	for i, construct := range constructs {
		fastaBuilder.WriteString(fmt.Sprintf(">Construct_%d\n", i+1))
		fastaBuilder.WriteString(construct)
		fastaBuilder.WriteString("\n")
	}

	return fastaBuilder.String()
}

func generateFastaOutputFromFragments(fragments []clone.Fragment) string {
	var fastaBuilder strings.Builder

	for i, fragment := range fragments {
		fastaBuilder.WriteString(fmt.Sprintf(">Fragment_%d\n", i+1))
		fastaBuilder.WriteString(fragment.Sequence)
		fastaBuilder.WriteString("\n")
	}

	return fastaBuilder.String()
}

func convertRebaseToCloneEnzyme(rebaseEnzyme rebase.Enzyme) (clone.Enzyme, error) {
	// Convert recognition sequence
	if strings.Contains(rebaseEnzyme.RecognitionSequence, "^") {
		parts := strings.Split(rebaseEnzyme.RecognitionSequence, "^")
		if len(parts) != 2 {
			return clone.Enzyme{}, fmt.Errorf("invalid recognition sequence format: %s", rebaseEnzyme.RecognitionSequence)
		}
		recSeq := parts[0] + parts[1]
		skip := -len(parts[1])
		overheadLength := 0

		regexpFor := regexp.MustCompile(recSeq)
		regexpRev := regexp.MustCompile(transform.ReverseComplement(recSeq))

		return clone.Enzyme{
			Name:            rebaseEnzyme.Name,
			RegexpFor:       regexpFor,
			RegexpRev:       regexpRev,
			Skip:            skip,
			OverheadLength:  overheadLength,
			RecognitionSite: recSeq,
		}, nil
	}

	// (n/m) format
	parts := strings.Split(rebaseEnzyme.RecognitionSequence, "(")
	if len(parts) != 2 {
		return clone.Enzyme{}, fmt.Errorf("invalid recognition sequence format: %s", rebaseEnzyme.RecognitionSequence)
	}
	recSeq := parts[0]
	cutPositions := strings.TrimRight(parts[1], ")")
	cutParts := strings.Split(cutPositions, "/")
	if len(cutParts) != 2 {
		return clone.Enzyme{}, fmt.Errorf("invalid cut positions format: %s", cutPositions)
	}
	forwardCut, err := strconv.Atoi(cutParts[0])
	if err != nil {
		return clone.Enzyme{}, fmt.Errorf("invalid forward cut position: %s", cutParts[0])
	}
	reverseCut, err := strconv.Atoi(cutParts[1])
	if err != nil {
		return clone.Enzyme{}, fmt.Errorf("invalid reverse cut position: %s", cutParts[1])
	}

	// Calculate skip and overheadLength
	skip := forwardCut
	overheadLength := reverseCut - forwardCut

	// Create regexp patterns
	regexpFor := regexp.MustCompile(recSeq)
	regexpRev := regexp.MustCompile(transform.ReverseComplement(recSeq))

	return clone.Enzyme{
		Name:            rebaseEnzyme.Name,
		RegexpFor:       regexpFor,
		RegexpRev:       regexpRev,
		Skip:            skip,
		OverheadLength:  overheadLength,
		RecognitionSite: recSeq,
	}, nil
}
