FROM ubuntu:14.04
COPY .setup_vm_and_run_tests.sh /
RUN chmod +x /.setup_vm_and_run_tests.sh
RUN ["/.setup_vm_and_run_tests.sh"]
