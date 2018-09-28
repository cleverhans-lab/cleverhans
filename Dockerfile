FROM ubuntu:14.04
COPY .setup_vm.sh /
COPY helloworld.sh /
RUN ["/helloworld.sh"]
CMD ["echo hello"]
