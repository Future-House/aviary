package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/bebop/poly/synthesis/codon"
	"github.com/bebop/poly/transform"
)

type ORF struct {
	Start       int    `json:"start"`
	End         int    `json:"end"`
	Strand      string `json:"strand"`
	Frame       int    `json:"frame"`
	Sequence    string `json:"sequence"`
	Translation string `json:"translation"`
}

// Modified FindORFs to include allowNested parameter
func FindORFs(sequence string, minLength int, codonTableIndex int, strand int, includeUnterminated bool, allowNested bool) []ORF {
	var orfs []ORF

	overwriteStartCodons := false
	if codonTableIndex == 0 {
		overwriteStartCodons = true
		codonTableIndex = 1
	}
	table, err := codon.NewTranslationTable(codonTableIndex)
	if err != nil {
		fmt.Printf("Error creating translation table: %v\n", err)
		return orfs
	}

	startCodons := table.StartCodons
	stopCodons := table.StopCodons

	// Overwrite startCodons to ["ATG"] if codonTableIndex == 0
	if overwriteStartCodons {
		startCodons = []string{"ATG"}
	}

	isStartCodon := func(c string) bool {
		for _, s := range startCodons {
			if c == s {
				return true
			}
		}
		return false
	}

	isStopCodon := func(c string) bool {
		for _, s := range stopCodons {
			if c == s {
				return true
			}
		}
		return false
	}

	findORFsInStrand := func(seq string, strandLabel string, frameOffset int) {
		for frame := 0; frame < 3; frame++ {
			// If not allowing nested ORFs, we track only one start at a time.
			// If allowing nested, we track multiple starts.
			var starts []int

			for i := frame; i < len(seq)-2; i += 3 {
				c := seq[i : i+3]
				if allowNested {
					// Nested mode: Each start codon opens a potential ORF
					if isStartCodon(c) {
						starts = append(starts, i)
					}
					// If stop codon, close all currently open starts
					if isStopCodon(c) && len(starts) > 0 {
						for _, st := range starts {
							orfSeq := seq[st : i+3]
							if len(orfSeq) >= minLength {
								translation, err := table.Translate(orfSeq)
								if err != nil {
									fmt.Printf("Error translating sequence: %v\n", err)
									continue
								}
								orfStart, orfEnd := st, i+2
								if strandLabel == "-" {
									orfStart, orfEnd = len(seq)-1-orfEnd, len(seq)-1-orfStart
								}
								orfs = append(orfs, ORF{
									Start:       orfStart,
									End:         orfEnd,
									Strand:      strandLabel,
									Frame:       frame + frameOffset + 1,
									Sequence:    orfSeq,
									Translation: translation,
								})
							}
						}
						// After encountering a stop, clear all starts
						starts = nil
					}
				} else {
					// Non-nested mode: Only one start at a time
					start := -1
					// We run a small loop here to mimic original logic
					// This will re-check codons as per original code.
					// We'll break out early once we find ORFs or continue scanning.
					start = -1
					for i2 := frame; i2 < len(seq)-2; i2 += 3 {
						cc := seq[i2 : i2+3]
						if start == -1 {
							if isStartCodon(cc) {
								start = i2
							}
						} else {
							if isStopCodon(cc) {
								orfSeq := seq[start : i2+3]
								if len(orfSeq) >= minLength {
									translation, err := table.Translate(orfSeq)
									if err != nil {
										fmt.Printf("Error translating sequence: %v\n", err)
										continue
									}
									orfStart, orfEnd := start, i2+2
									if strandLabel == "-" {
										orfStart, orfEnd = len(seq)-1-orfEnd, len(seq)-1-orfStart
									}
									orfs = append(orfs, ORF{
										Start:       orfStart,
										End:         orfEnd,
										Strand:      strandLabel,
										Frame:       frame + frameOffset + 1,
										Sequence:    orfSeq,
										Translation: translation,
									})
								}
								start = -1
							}
						}
					}
					// If including unterminated ORFs in non-nested mode
					if includeUnterminated && start != -1 {
						orfSeq := seq[start:]
						if len(orfSeq) >= minLength {
							translation, err := table.Translate(orfSeq)
							if err != nil {
								fmt.Printf("Error translating sequence: %v\n", err)
							} else {
								orfStart, orfEnd := start, len(seq)-1
								if strandLabel == "-" {
									orfStart, orfEnd = len(seq)-1-orfEnd, len(seq)-1-orfStart
								}
								orfs = append(orfs, ORF{
									Start:       orfStart,
									End:         orfEnd,
									Strand:      strandLabel,
									Frame:       frame + frameOffset + 1,
									Sequence:    orfSeq,
									Translation: translation,
								})
							}
						}
					}
					// Move to next frame
					break
				}
			}

			// If including unterminated and we allow nested:
			if allowNested && includeUnterminated && len(starts) > 0 {
				for _, st := range starts {
					orfSeq := seq[st:]
					if len(orfSeq) >= minLength {
						translation, err := table.Translate(orfSeq)
						if err != nil {
							fmt.Printf("Error translating sequence: %v\n", err)
							continue
						}
						orfStart, orfEnd := st, len(seq)-1
						if strandLabel == "-" {
							orfStart, orfEnd = len(seq)-1-orfEnd, len(seq)-1-orfStart
						}
						orfs = append(orfs, ORF{
							Start:       orfStart,
							End:         orfEnd,
							Strand:      strandLabel,
							Frame:       frame + frameOffset + 1,
							Sequence:    orfSeq,
							Translation: translation,
						})
					}
				}
			}

			// If we are in non-nested mode, we've already handled frames above.
			if !allowNested {
				// break out of the frame loop since we processed them inside the nested loop
				return
			}
		}
	}

	if strand == 1 || strand == 0 {
		findORFsInStrand(sequence, "+", 0)
	}
	if strand == -1 || strand == 0 {
		revComp := transform.ReverseComplement(sequence)
		findORFsInStrand(revComp, "-", 3)
	}

	return orfs
}

// parseFasta reads a FASTA file and returns a single sequence.
func parseFasta(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open FASTA file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var sequences []string
	var currentSeq strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, ">") {
			if currentSeq.Len() > 0 {
				sequences = append(sequences, currentSeq.String())
				currentSeq.Reset()
			}
		} else {
			currentSeq.WriteString(strings.TrimSpace(line))
		}
	}

	if currentSeq.Len() > 0 {
		sequences = append(sequences, currentSeq.String())
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading FASTA file: %v", err)
	}

	if len(sequences) != 1 {
		return "", fmt.Errorf("FASTA file must contain exactly one sequence, found %d", len(sequences))
	}

	return sequences[0], nil
}

func main() {
	fastaPath := flag.String("fasta", "", "Path to input FASTA file (required)")
	minLength := flag.Int("min-length", 300, "Minimum ORF length in base pairs")
	codonTableIndex := flag.Int("codon-table", 1, "Codon table index (default: 1 for standard)")
	strand := flag.Int("strand", 0, "Strand to search (0: both, 1: forward, -1: reverse)")
	includeUnterminated := flag.Bool("include-unterminated", false, "Include unterminated ORFs")
	allowNested := flag.Bool("allow-nested", false, "Allow nested (overlapping) ORFs")

	flag.Parse()

	if *fastaPath == "" {
		fmt.Println("Error: Path to input FASTA file is required")
		flag.Usage()
		os.Exit(1)
	}

	sequence, err := parseFasta(*fastaPath)
	if err != nil {
		fmt.Printf("Error parsing FASTA file: %v\n", err)
		os.Exit(1)
	}

	upperSequence := strings.ToUpper(sequence)

	orfs := FindORFs(upperSequence, *minLength, *codonTableIndex, *strand, *includeUnterminated, *allowNested)

	// Sort ORFs by protein length (descending)
	sort.Slice(orfs, func(i, j int) bool {
		return len(orfs[i].Translation) > len(orfs[j].Translation)
	})

	result := struct {
		Sequence        string `json:"sequence"`
		MinLength       int    `json:"minLength"`
		CodonTableIndex int    `json:"codonTableIndex"`
		Strand          int    `json:"strand"`
		AllowNested     bool   `json:"allowNested"`
		ORFs            []ORF  `json:"orfs"`
	}{
		Sequence:        upperSequence,
		MinLength:       *minLength,
		CodonTableIndex: *codonTableIndex,
		Strand:          *strand,
		AllowNested:     *allowNested,
		ORFs:            orfs,
	}

	jsonOutput, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Error encoding to JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(jsonOutput))
}
