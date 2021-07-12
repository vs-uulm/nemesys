FROM python:3.8-buster

WORKDIR /nemere

# install missing system deps
# tshark installation includes interactive question, thus using DEBIAN_FRONTEND var to prevent this
RUN echo 'wireshark-common wireshark-common/install-setuid boolean true' | debconf-set-selections -
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    libpcap-dev \
    tshark

# install required python modules
COPY /requirements.txt .
RUN pip3 install -r requirements.txt
# pylstar should be installed by netzob, but pip fails to install it there
RUN pip3 install pylstar==0.1.2

# install netzob
# (tested on commit https://github.com/netzob/netzob/tree/49ee3e5e7d6dce67496afd5a75827a78be0c9f70)
RUN git clone --single-branch -b next https://github.com/netzob/netzob.git
RUN cd netzob/netzob && \
    git checkout 63125dbd31d28c27eee8616bd21345af417f5310 && \
    python3 setup.py install && \
    cd ../..

# copy nemere
COPY . .

# create user and add them to wireshark group
RUN useradd -ms /bin/bash user
RUN gpasswd -a user wireshark
USER user

# start in blank shell
CMD [ "/bin/bash" ]
