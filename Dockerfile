# Use Node 18
FROM node:18-slim

# Set the working directory
WORKDIR /app

# Step 1: Check what files actually exist (for debugging)
COPY . .
RUN ls -la

# Step 2: Try to install
# We use --force because you don't have a lockfile
RUN npm install --no-audit --force

# Step 3: Build
RUN npm run build

# Step 4: Start
EXPOSE 3000
CMD ["npm", "start"]
