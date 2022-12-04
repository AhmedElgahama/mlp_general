FROM r-base:4.2.2

RUN apt-get update && \
  apt-get install -y libxml2-dev

WORKDIR /opt/app

# Install binaries (see https://datawookie.netlify.com/blog/2019/01/docker-images-for-r-r-base-versus-r-apt/)
COPY ./requirements-bin.txt .
RUN cat requirements-bin.txt | xargs apt-get install -y -qq


# Install remaining packages from source (these dont have binaries)
COPY ./requirements-src.R .
RUN Rscript requirements-src.R

# Clean up package registry
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY app ./opt/app
WORKDIR /opt/app


ENV PATH="/opt/app:${PATH}"

RUN chmod +x train
RUN chmod +x predict
RUN chmod +x serve

# USER 1001


