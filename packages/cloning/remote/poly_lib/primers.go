package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/bebop/poly/primers/pcr"
)

type Result struct {
	ForwardPrimer string `json:"forward_primer"`
	ReversePrimer string `json:"reverse_primer"`
	AmpliconFasta string `json:"amplicon_fasta"`
}

func main() {
	// Define command-line flags
	sequence := flag.String("sequence", "", "Input DNA sequence")
	forwardOverhang := flag.String("forward-overhang", "", "Forward primer overhang")
	reverseOverhang := flag.String("reverse-overhang", "", "Reverse primer overhang")
	targetTm := flag.Float64("target-tm", 90.0, "Target melting temperature for primers")
	forwardPrimerSeq := flag.String("forward-primer", "", "Forward primer sequence")
	reversePrimerSeq := flag.String("reverse-primer", "", "Reverse primer sequence")
	circular := flag.Bool("circular", false, "Is the input sequence circular?")

	flag.Parse()

	// Validate input
	if *sequence == "" {
		fmt.Println("Error: Input DNA sequence is required")
		flag.Usage()
		os.Exit(1)
	}

	// Check if primers are provided
	var forwardPrimer, reversePrimer string
	if *forwardPrimerSeq != "" && *reversePrimerSeq != "" {
		// Use provided primers
		forwardPrimer = strings.ToUpper(*forwardPrimerSeq)
		reversePrimer = strings.ToUpper(*reversePrimerSeq)
	} else if *forwardPrimerSeq == "" && *reversePrimerSeq == "" {
		// Design primers
		forwardPrimer, reversePrimer = pcr.DesignPrimersWithOverhangs(*sequence, *forwardOverhang, *reverseOverhang, *targetTm)
	} else {
		// Error if only one primer is provided
		fmt.Println("Error: Both forward and reverse primers must be provided")
		flag.Usage()
		os.Exit(1)
	}

	// Ensure primers meet the minimal length requirement
	minimalPrimerLength := 7 // As defined in the pcr package
	if len(forwardPrimer) < minimalPrimerLength {
		fmt.Printf("Error: Forward primer must be at least %d nucleotides long\n", minimalPrimerLength)
		os.Exit(1)
	}
	if len(reversePrimer) < minimalPrimerLength {
		fmt.Printf("Error: Reverse primer must be at least %d nucleotides long\n", minimalPrimerLength)
		os.Exit(1)
	}

	// Simulate PCR
	fragments := pcr.SimulateSimple([]string{*sequence}, *targetTm, *circular, []string{forwardPrimer, reversePrimer})
	// TODO: revisit the below. Unclear if we should throw an error on concatemerization (the only error that can be thrown)
	// or if we can instead pass a warning to the agent somehow.
	// fragments, err := pcr.Simulate([]string{*sequence}, *targetTm, *circular, []string{forwardPrimer, reversePrimer})
	// if err != nil {
	// 	fmt.Printf("Error during PCR simulation: %v\n", err)
	// 	// print out inputs to simulations
	// 	fmt.Printf("sequence: %s\n", *sequence)
	// 	fmt.Printf("targetTm: %f\n", *targetTm)
	// 	fmt.Printf("circular: %t\n", *circular)
	// 	fmt.Printf("primers: %s, %s\n", forwardPrimer, reversePrimer)
	// 	os.Exit(1)
	// }

	fastaOutput := generateFastaOutputFromFragments(fragments)

	// Prepare result
	result := Result{
		ForwardPrimer: forwardPrimer,
		ReversePrimer: reversePrimer,
		AmpliconFasta: fastaOutput,
	}

	// Output result
	jsonOutput, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Error encoding to JSON: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(jsonOutput))
}

func generateFastaOutputFromFragments(fragments []string) string {
	var fastaBuilder strings.Builder

	for i, fragment := range fragments {
		fastaBuilder.WriteString(fmt.Sprintf(">Amplicon_%d\n", i+1))
		fastaBuilder.WriteString(fragment)
		fastaBuilder.WriteString("\n")
	}

	return fastaBuilder.String()
}
