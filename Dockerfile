FROM ubuntu:14.04
COPY .setup_vm.sh /
RUN chmod +x /.setup_vm.sh
RUN ["/.setup_vm.sh"]
COPY .run_tests.sh /
RUN chmod +x /.run_tests.sh
RUN ["/.run_tests.sh"]
