package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/bebop/poly/io/fasta"
	"github.com/bebop/poly/seqhash"
	"github.com/bebop/poly/transform"
)

/*
	Gibson Assembly Simulation Tool

	This program simulates the Gibson Assembly method for joining multiple DNA fragments seamlessly.
	It reads DNA sequences from a FASTA file, identifies optimal overlaps between fragments (including reverse complements),
	and assembles them into longer sequences, detecting both linear and circular assemblies.

*/

// Fragment represents a DNA sequence fragment with its associated metadata.
type Fragment struct {
	Name       string // Identifier for the fragment
	Sequence   string // DNA sequence of the fragment
	IsCircular bool   // Indicates if the assembled sequence is circular
}

// Overlap represents the overlap information between two fragments.
type Overlap struct {
	Length      int    // Length of the overlap
	Sequence    string // Overlapping sequence
	Orientation string // "forward" or "reverse"
}

// AssemblyResult represents a successfully assembled sequence.
type AssemblyResult struct {
	Sequence   string   // Assembled DNA sequence
	IsCircular bool     // Indicates if the sequence is circular
	Fragments  []string // Names of fragments used in the assembly
}

func main() {
	// Parse command-line arguments
	fastaFile := flag.String("fasta", "", "Path to the input FASTA file")
	flag.Parse()

	if *fastaFile == "" {
		fmt.Println("Usage: ./gibson-cli -fasta <path_to_fasta_file>")
		os.Exit(1)
	}

	// Read and validate fragments from the FASTA file
	fragments, err := readFastaFile(*fastaFile)
	if err != nil {
		log.Fatalf("Error reading FASTA file: %v", err)
	}

	// Simulate Gibson Assembly
	assemblies := simulateAssembly(fragments)

	// Print assembly results
	printAssemblyResults(assemblies)
}

// readFastaFile reads DNA sequences from a FASTA file and validates them.
func readFastaFile(path string) ([]Fragment, error) {
	fastaEntries, err := fasta.Read(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read FASTA file: %w", err)
	}

	// we depend on names being unique
	names := make(map[string]bool)

	var fragments []Fragment
	for _, entry := range fastaEntries {
		sequence := strings.ToUpper(entry.Sequence)

		isCircular := strings.Contains(strings.ToLower(entry.Name), "(circular)")

		// throw error if sequence is circular
		if isCircular {
			return nil, fmt.Errorf("circular sequences are not supported: %s", entry.Name)
		}

		// Ensure fragment names are unique
		i := 0
		for names[entry.Name] {
			i++
			entry.Name = fmt.Sprintf("%s_%d", entry.Name, i)
		}
		names[entry.Name] = true
		fragments = append(fragments, Fragment{
			Name:     entry.Name,
			Sequence: sequence,
		})
	}

	return fragments, nil
}

// simulateAssembly performs Gibson Assembly simulation on the provided fragments.
func simulateAssembly(fragments []Fragment) []AssemblyResult {
	var assemblies []AssemblyResult

	// Start assembly with each fragment as the initial seed
	for _, fragment := range fragments {
		used := map[string]bool{fragment.Name: true}
		currentPath := []string{fragment.Name}
		results := assembleFragments(fragment, fragments, used, currentPath, true)
		assemblies = append(assemblies, results...)
	}

	return assemblies
}

// assembleFragments recursively assembles fragments based on overlaps.
func assembleFragments(current Fragment, fragments []Fragment, used map[string]bool, path []string, top bool) []AssemblyResult {

	// top = non-recursive call

	var results []AssemblyResult
	hasExtension := false

	for _, fragment := range fragments {
		if used[fragment.Name] {
			continue
		}

		// Check for overlaps in both orientations
		overlap, success := findMaxOverlap(current.Sequence, fragment.Sequence)
		if success {
			hasExtension = true
			newSequence := current.Sequence + fragment.Sequence[overlap.Length:]
			newUsed := cloneUsedMap(used)
			newUsed[fragment.Name] = true
			newPath := append(path, fragment.Name)

			extendedFragment := Fragment{
				Name:     current.Name + "_" + fragment.Name,
				Sequence: newSequence,
			}

			// Recursively extend the assembly
			subResults := assembleFragments(extendedFragment, fragments, newUsed, newPath, false)
			results = append(results, subResults...)
		}

		// Check for overlaps with reverse complement
		revCompSeq := transform.ReverseComplement(fragment.Sequence)
		overlapRC, successRC := findMaxOverlap(current.Sequence, revCompSeq)
		if successRC {
			hasExtension = true
			newSequence := current.Sequence + revCompSeq[overlapRC.Length:]
			newUsed := cloneUsedMap(used)
			newUsed[fragment.Name] = true
			newPath := append(path, fragment.Name+"(rev_comp)")

			extendedFragment := Fragment{
				Name:     current.Name + "_" + fragment.Name + "_rc",
				Sequence: newSequence,
			}

			// Recursively extend the assembly
			subResults := assembleFragments(extendedFragment, fragments, newUsed, newPath, false)
			results = append(results, subResults...)
		}
	}

	// If no further extension is possible, check for circularization
	// Note - this is the tail where we return the passed fragment as the result
	// if top is true, we do not need to return/check for circularization
	if !hasExtension && !top {
		isCircular, circularSequence := checkCircularization(current.Sequence)
		results = append(results, AssemblyResult{
			Sequence:   circularSequence,
			IsCircular: isCircular,
			Fragments:  path,
		})
	}

	return results
}

// findMaxOverlap finds the maximum overlap between two sequences.
func findMaxOverlap(seq1, seq2 string) (Overlap, bool) {
	minOverlapLength := 10 // Minimum acceptable overlap length
	maxOverlap := Overlap{Length: 0}

	// Determine the shorter sequence length
	maxPossibleLength := min(len(seq1), len(seq2))

	// Search for maximum overlap where seq1 suffix matches seq2 prefix
	for length := maxPossibleLength; length >= minOverlapLength; length-- {
		if seq1[len(seq1)-length:] == seq2[:length] {
			maxOverlap = Overlap{
				Length:      length,
				Sequence:    seq1[len(seq1)-length:],
				Orientation: "forward",
			}
			return maxOverlap, true
		}
	}

	return maxOverlap, false
}

// checkCircularization checks if a sequence can be circularized based on its overlapping ends.
func checkCircularization(sequence string) (bool, string) {
	minOverlapLength := 10 // Minimum required overlap length for circularization
	maxPossibleLength := len(sequence) / 2

	for length := maxPossibleLength; length >= minOverlapLength; length-- {
		if sequence[:length] == sequence[len(sequence)-length:] {
			// Circularization achieved by removing redundant overlap
			circularSequence := sequence[:len(sequence)-length]
			return true, circularSequence
		}
	}

	// No sufficient overlap for circularization
	return false, sequence
}

// cloneUsedMap creates a copy of the used fragments map.
func cloneUsedMap(original map[string]bool) map[string]bool {
	clone := make(map[string]bool)
	for key, value := range original {
		clone[key] = value
	}
	return clone
}

// printAssemblyResults outputs the assembly results in FASTA format.
func printAssemblyResults(results []AssemblyResult) {
	uniqueSequences := make(map[string]bool)
	for i, result := range results {
		// Avoid printing duplicate sequences
		seqHash, _ := seqhash.Hash(result.Sequence, "DNA", true, true)
		if uniqueSequences[seqHash] {
			continue
		}
		uniqueSequences[seqHash] = true

		description := fmt.Sprintf("Assembly_%d | Fragments: %s", i+1, strings.Join(result.Fragments, " -> "))
		// remove linear or circular from description before appending
		re := regexp.MustCompile(`\s*\((linear|circular)\)\s*`)
		description = re.ReplaceAllString(description, "")
		if result.IsCircular {
			description += " | (circular)"
		} else {
			description += " | (linear)"
		}
		fmt.Printf(">%s\n%s\n\n", description, formatSequence(result.Sequence, 80))
	}
}

// formatSequence formats a DNA sequence into lines of specified length.
func formatSequence(sequence string, lineLength int) string {
	var builder strings.Builder
	for i := 0; i < len(sequence); i += lineLength {
		end := i + lineLength
		if end > len(sequence) {
			end = len(sequence)
		}
		builder.WriteString(sequence[i:end] + "\n")
	}
	return builder.String()
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
