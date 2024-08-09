FROM node:lts

# Set the working directory
WORKDIR /usr/src/app

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Copy package.json and package-lock.json first to leverage Docker cache
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Define build argument
ARG PROFILE
ENV PROFILE $PROFILE

# Display environment for debugging
RUN echo "Environment: ${PROFILE}"

# Expose port
EXPOSE 4200

# Set entrypoint for different environments
ENTRYPOINT bash -c "if [ \"$PROFILE\" = \"production\" ]; then ng serve --host 0.0.0.0 --port 4200 --configuration=production --disable-host-check; elif [ \"$PROFILE\" = \"docker\" ]; then ng serve --host 0.0.0.0 --port 4200 --configuration=docker --disable-host-check; else ng serve --host 0.0.0.0 --port 4200 --disable-host-check; fi"
