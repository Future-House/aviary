# This script checks if IRKernel is installed, installs it if not, and runs installspec.

# make sure R_LIBS_USER exists
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
# install for user
install.packages("IRkernel", lib = Sys.getenv("R_LIBS_USER"))
