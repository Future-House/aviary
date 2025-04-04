package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"

	"github.com/bebop/poly/checks"
	"github.com/bebop/poly/synthesis/codon"
	"github.com/bebop/poly/synthesis/fix"
)

// We use snake_case to line-up with pydantic klater
type OptimizationResult struct {
	InputSequence      string       `json:"input_sequence"`
	InputType          string       `json:"input_type"`
	OptimizedDNA       string       `json:"optimized_dna"`
	Changes            []fix.Change `json:"changes"`
	InitialGCContent   float64      `json:"initial_gc_content"`
	OptimizedGCContent float64      `json:"optimized_gc_content"`
	CodonTableIndex    int          `json:"codon_table_index"`
}

func main() {
	// Define command-line flags
	inputSequence := flag.String("sequence", "", "Input sequence (DNA or amino acids)")
	inputType := flag.String("type", "", "Input type: 'dna' or 'aa'")
	maxGCContent := flag.Float64("max-gc", 62.0, "Maximum GC content percentage")
	repeatLength := flag.Int("repeat-length", 15, "Minimum length of repeats to remove")
	codonTableIndex := flag.Int("codon-table", 11, "Codon table index (default: 11 for bacterial)")
	flag.Parse()

	// Validate input
	if *inputSequence == "" || *inputType == "" {
		fmt.Println("Error: Input sequence and type are required")
		flag.Usage()
		os.Exit(1)
	}

	if *inputType != "dna" && *inputType != "protein" {
		fmt.Println("Error: Input type must be 'dna' or 'protein'")
		flag.Usage()
		os.Exit(1)
	}

	// Create codon table
	codonTable, err := codon.NewTranslationTable(*codonTableIndex)
	if err != nil {
		fmt.Printf("Error creating translation table: %v\n", err)
		os.Exit(1)
	}

	// Initialize DNA sequence
	var initialDNA string
	if *inputType == "protein" {
		// Optimize codons if input is amino acids
		initialDNA, err = codonTable.Optimize(*inputSequence)
		if err != nil {
			fmt.Printf("Error optimizing codons: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Use input directly if it's DNA
		initialDNA = *inputSequence
	}

	// Define optimization functions
	var optimizationFuncs []func(string, chan fix.DnaSuggestion, *sync.WaitGroup)

	// GC content fixer
	optimizationFuncs = append(optimizationFuncs, fix.GcContentFixer(*maxGCContent/100, 0))

	// Repeat remover
	optimizationFuncs = append(optimizationFuncs, fix.RemoveRepeat(*repeatLength))

	// Optimize sequence
	optimizedDNA, changes, err := fix.Cds(initialDNA, codonTable, optimizationFuncs)
	if err != nil {
		fmt.Printf("Error optimizing sequence: %v\n", err)
		os.Exit(1)
	}

	// Calculate GC content
	initialGCContent := checks.GcContent(initialDNA)
	optimizedGCContent := checks.GcContent(optimizedDNA)

	// Verify optimization
	if *inputType == "protein" {
		optimizedAA, err := codonTable.Translate(optimizedDNA)
		if err != nil {
			fmt.Printf("Error translating optimized DNA: %v\n", err)
			os.Exit(1)
		}
		if optimizedAA != *inputSequence {
			fmt.Println("Error: Optimized DNA does not translate back to the original amino acid sequence")
			os.Exit(1)
		}
	}

	// Prepare result
	result := OptimizationResult{
		InputSequence:      *inputSequence,
		InputType:          *inputType,
		OptimizedDNA:       optimizedDNA,
		Changes:            changes,
		InitialGCContent:   initialGCContent * 100,
		OptimizedGCContent: optimizedGCContent * 100,
		CodonTableIndex:    *codonTableIndex,
	}

	// Output result
	jsonOutput, err := json.MarshalIndent(result, "", " ")
	if err != nil {
		fmt.Printf("Error encoding to JSON: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(jsonOutput))
}
