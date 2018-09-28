FROM ubuntu:14.04
COPY .setup_vm.sh /
RUN ["/.setup_vm.sh"]
COPY helloworld.sh /
CMD ["/helloworld.sh"]
