FROM ubuntu:14.04
COPY .setup_vm.sh /
COPY .run_tests.sh /
CMD ["/.setup_vm.sh"]
CMD ["/.run_tests.sh"]
