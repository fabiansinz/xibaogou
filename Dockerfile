FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.edu>

# Build HDF5
RUN cd ; wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz \
    && tar zxf hdf5-1.8.16.tar.gz \
    && mv hdf5-1.8.16 hdf5-setup \
    &&  cd hdf5-setup \
    && ./configure --prefix=/usr/local/ \
    &&  make -j 12 && make install \
    && cd  \
    && rm -rf hdf5-setup \
    && apt-get -yq autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install HDF5 reader
RUN pip install h5py

# Get pupil tracking repo
RUN \
  git clone https://github.com/fabiansinz/xibaogou.git && \
  pip install -e xibaogou/

ENTRYPOINT /bin/bash
