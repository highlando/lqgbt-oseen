FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:fenics-packages/fenics -y
RUN apt-get install --no-install-recommends fenics -y
RUN apt-get install build-essential -y
RUN apt install openssh-server -y
RUN dolfin-get-demos -y
RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN pip3 install mat73

# Dann noch installieren:
# sudo apt-get update
# sudo apt-get -y install python3-pip
