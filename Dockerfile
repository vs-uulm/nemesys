FROM python:3.9-bookworm

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

# copy nemere
COPY . .

# create user and add them to wireshark group
RUN useradd -ms /bin/bash user
RUN gpasswd -a user wireshark
USER user

# start in blank shell
CMD [ "/bin/bash" ]

