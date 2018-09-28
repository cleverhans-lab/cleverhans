FROM ubuntu:14.04
COPY .setup_vm.sh /
COPY .run_tests.sh /
RUN ["/.setup_vm.sh"]
RUN ["/.run_tests.sh"]
