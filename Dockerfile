# Save at the root of CanDIAPL/AstroPhot-tool, beside pyproject.toml.
# A small command-line worker: install the package, read a per-run JSON config,
# call joint_multi_image(), write JSON results, and exit. 

# Python 3.12 satisfies the repository's Python >=3.10 requirement.
# For a reproducible release, pin a tested image digest and Python dependencies.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/astrophot-tool

# AstroPhot uses PyTorch. Install CPU wheels first for this lightweight CPU 
# starter. 
RUN python -m pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Copy only build inputs and code. Science inputs, credentials, configs, and
# results are supplied at runtime and should never be baked into this image.
COPY pyproject.toml LICENSE ./
COPY src/ ./src/
COPY container_entrypoint.py ./container_entrypoint.py

# This installs the local package and its declared imports: astrophot, astropy,
# numpy, scipy (and their dependencies). Keep dependencies in pyproject.toml.
# For a non-packaged tool instead: COPY requirements.txt ./ then pip install
# -r requirements.txt; copy its script(s) and update ENTRYPOINT accordingly.
RUN python -m pip install --no-cache-dir . \
    && python -c "from AstroPhot_tool import joint_multi_image"

# Some AstroPhot versions create AstroPhot.log in the current directory on
# import. Runtime work belongs in writable temporary space, not the code tree.
# Inputs and results use explicit paths, so this does not move science outputs.
WORKDIR /tmp

# Local Docker runs default to a non-root user. CANFAR supplies the submitting
# user's UID/groups, so code and installed packages must remain world-readable.
# The chosen output directory must be writable by the runtime user.
USER 10001:10001

# ENTRYPOINT runs when a container starts, not when this Dockerfile is built.
# Extra docker-run arguments are appended to this command. On CANFAR, a job
# launcher may instead invoke the same absolute script path explicitly.
ENTRYPOINT ["python", "/opt/astrophot-tool/container_entrypoint.py"]

# Running the image with no arguments shows usage rather than starting a fit.
CMD ["--help"]
