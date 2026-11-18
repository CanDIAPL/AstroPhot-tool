# Save at the root of CanDIAPL/AstroPhot-tool, beside pyproject.toml.
# A small command-line worker: install the package, read a per-run JSON config,
# call joint_multi_image(), write JSON results, and exit. 

# Python 3.12 is inside the >=3.11,<3.14 range of pyproject.toml and is the
# version harbor.yml tests on. The tag names an exact Python release and Debian
# version, so a rebuild starts from the same base. (A tag can still be
# re-pushed upstream; "FROM python:...@sha256:<digest>" is the strictest form.)
FROM python:3.12.16-slim-trixie

# HOME: the runtime user below has no account, and astropy needs a home
# directory to exist. A platform that sets its own HOME overrides this.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

WORKDIR /opt/astrophot-tool

# Copy only build inputs and code. Science inputs, credentials, configs, and
# results are supplied at runtime and should never be baked into this image.
COPY pyproject.toml LICENSE ./
COPY src/ ./src/
COPY container_entrypoint.py ./container_entrypoint.py

# This installs the local package and the exact versions pinned in
# pyproject.toml (astrophot, astropy, numpy, scipy, torch, caskade) with their
# dependencies. Keep dependencies in pyproject.toml.
# AstroPhot uses PyTorch: the extra index supplies the CPU only build of the
# pinned torch version, the default build would add gigabytes of CUDA
# libraries. The first check fails the build if that ever stops working.
# The second check imports the package, as root. That import writes
# AstroPhot.log here and a matplotlib font cache in MPLCONFIGDIR: the log is
# removed and the cache is opened up so any runtime user can reuse it.
# The package version normally comes from the git tag, but the git history is
# not copied in here, so it is given as a build argument (harbor.yml passes
# the release tag): docker build --build-arg VERSION=0.2.0 .
ARG VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" python -m pip install --no-cache-dir . \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__" \
    && python -c "from AstroPhot_tool import joint_multi_image" \
    && rm -f AstroPhot.log \
    && chmod -R a+rwX "$MPLCONFIGDIR"

# AstroPhot creates AstroPhot.log in the current directory on import. Runtime
# work belongs in writable temporary space, not the code tree. Inputs and
# results use explicit paths, so this does not move science outputs. (The
# entrypoint also moves to a temporary directory itself, for a platform that
# starts the container somewhere else.)
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
