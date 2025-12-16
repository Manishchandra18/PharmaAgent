#!/bin/bash

set -xe

# Install build tools needed for packages like PyMuPDF (fitz) that compile native code
if command -v dnf >/dev/null 2>&1; then
  dnf install -y gcc gcc-c++ make
else
  yum install -y gcc gcc-c++ make
fi


