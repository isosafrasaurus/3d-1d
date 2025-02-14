FROM ghcr.io/isosafrasaurus/fem-on-jupyter/fenics:latest

RUN git clone "https://bitbucket.org/fenics-apps/cbc.block/src/master/" /opt/block && \
    pip install /opt/block

RUN git clone --single-branch -b "collapse-iter-dev" "https://github.com/MiroK/fenics_ii" /opt/fenics_ii && \
    find /opt/fenics_ii -type f -name "*.py" -exec sed -i 's/\bufl\b/ufl_legacy/g' {} \; && \
    pip install /opt/fenics_ii

RUN git clone "https://github.com/IngeborgGjerde/graphnics" /opt/graphnics && \
    pip install /opt/graphnics

COPY . /root
