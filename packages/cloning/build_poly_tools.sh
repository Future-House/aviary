#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# =======================
# Configuration Variables
# =======================

# Use location of the script to determine root cloning directory
CLONING_DIR_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REMOTE_DIR="$CLONING_DIR_PATH/remote"
OUTPUT_DIR="$CLONING_DIR_PATH/cloning/bin"

# Default build settings
DEFAULT_GOOS="linux"
DEFAULT_GOARCH="amd64"

# Command-line overrides
GOOS=${1:-$DEFAULT_GOOS}
GOARCH=${2:-$DEFAULT_GOARCH}

# =======================
# Function Definitions
# =======================

# Function to set up Go dependencies
setup_go_dependencies() {
    echo "Setting up Go dependencies..."

    # Create the remote directory if it doesn't exist
    mkdir -p "$REMOTE_DIR"
    cd "$REMOTE_DIR"

    # Initialize Go module if go.mod does not exist
    if [ ! -f go.mod ]; then
        echo "Initializing Go module..."
        go mod init poly_lib
    else
        echo "Go module already initialized."
    fi

    # Replace the specific dependency as per GitHub Actions
    echo "Replacing dependency github.com/bebop/poly with github.com/whitead/poly@f92359b10f3e57a9712f5fe5b2ccd0d78154fe76..."
    go mod edit -replace=github.com/bebop/poly=github.com/whitead/poly@f92359b10f3e57a9712f5fe5b2ccd0d78154fe76

    # Install required Go packages
    echo "Installing required Go packages..."
    go get github.com/bebop/poly/synthesis/codon
    go get github.com/bebop/poly/transform
    go get github.com/bebop/poly/synthesis/fix
    go get github.com/bebop/poly/checks
    go get github.com/bebop/poly/clone
    go get github.com/bebop/poly/primers/pcr
    go get github.com/bebop/poly/io/fasta

    echo "Go dependencies installed successfully."
}

# Function to build the executables
build_executables() {
    # Create the output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    cd "$REMOTE_DIR"

    # Set Go environment variables for cross-compilation if needed
    export GOOS="$GOOS"
    export GOARCH="$GOARCH"

    echo "Building executables for GOOS=$GOOS and GOARCH=$GOARCH..."

    # Build the clone binary
    for program in orf clone primers gibson synthesize; do
        echo "Compiling ${program}.go into binary..."
        go build -o "$OUTPUT_DIR/$program" "./poly_lib/${program}.go"
        echo "Successfully built $OUTPUT_DIR/$program"
    done
}

# =======================
# Main Execution Flow
# =======================

main() {
    echo "Starting build process with GOOS=$GOOS and GOARCH=$GOARCH..."
    setup_go_dependencies
    build_executables
    echo "Build process completed successfully."
}

# Invoke the main function
main
