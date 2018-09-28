FROM alpine
COPY helloworld.sh /
RUN ["/helloworld.sh"]
CMD ["echo hello"]
