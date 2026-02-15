# syntax=docker/dockerfile:1
FROM node:18-slim AS base

WORKDIR /app

# Install dependencies with extra flags to prevent timeouts
COPY package.json ./
RUN npm install --no-audit --prefer-offline

# Copy the rest of the code
COPY . .

# Run the Next.js build
RUN npm run build

# Start the app
CMD ["npm", "start"]
