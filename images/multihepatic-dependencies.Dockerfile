FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

USER root

ARG USERNAME=dolfinx
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create/align a non-root user reliably:
# - If a group with USER_GID exists, reuse it.
# - If a user with USER_UID exists, reuse it (and optionally rename it).
# - Otherwise create the requested user/group.
RUN set -eux; \
    # Figure out group name for requested GID (create if missing) \
    if getent group "${USER_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        GROUP_NAME="${USERNAME}"; \
        groupadd --gid "${USER_GID}" "${GROUP_NAME}"; \
    fi; \
    \
    # Figure out username for requested UID (create if missing) \
    if getent passwd "${USER_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        USER_NAME="${EXISTING_USER}"; \
        # If caller wants a specific username, rename the existing UID owner to that name (safe if name unused) \
        if [ "${EXISTING_USER}" != "${USERNAME}" ] && ! getent passwd "${USERNAME}" >/dev/null; then \
            usermod -l "${USERNAME}" "${EXISTING_USER}"; \
            usermod -d "/home/${USERNAME}" -m "${USERNAME}"; \
            USER_NAME="${USERNAME}"; \
        fi; \
        # Ensure the user is in the desired primary group \
        usermod -g "${GROUP_NAME}" "${USER_NAME}"; \
    else \
        USER_NAME="${USERNAME}"; \
        # If the username exists but with a different UID, adjust it; otherwise create it \
        if getent passwd "${USERNAME}" >/dev/null; then \
            usermod -u "${USER_UID}" -g "${GROUP_NAME}" "${USERNAME}"; \
        else \
            useradd -m -u "${USER_UID}" -g "${GROUP_NAME}" -s /bin/bash "${USERNAME}"; \
        fi; \
    fi; \
    \
    # Ensure home exists and ownership is correct \
    mkdir -p "/home/${USERNAME}"; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

WORKDIR /workspace

ENV HOME=/home/${USERNAME}
ENV XDG_CACHE_HOME=/home/${USERNAME}/.cache
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/home/${USERNAME}/.cache/matplotlib

RUN mkdir -p /tmp/src \
    /home/${USERNAME}/.cache \
    /home/${USERNAME}/.cache/fenics \
    /workspace && \
    chown -R ${USER_UID}:${USER_GID} /home/${USERNAME} /workspace /tmp/src

# Keep your system site-packages pointer logic (adjust python version if needed)
RUN SYSTEM_SITE_PACKAGES=$(python3 -c "import sys; print([p for p in sys.path if 'dist-packages' in p and 'local' in p][0])") && \
    echo "Found system packages at: $SYSTEM_SITE_PACKAGES" && \
    echo "$SYSTEM_SITE_PACKAGES" > /dolfinx-env/lib/python3.12/site-packages/system_packages.pth

RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/fenicsx_ii.git && \
    cd fenicsx_ii && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/networks_fenicsx.git && \
    cd networks_fenicsx && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

RUN /dolfinx-env/bin/python3 -m pip install \
    networkx vtk meshio nibabel pyvista tetgen pymeshfix h5py

RUN rm -rf /tmp/src

USER ${USERNAME}
