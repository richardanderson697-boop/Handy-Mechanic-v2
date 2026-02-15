# Use a lightweight Node image
FROM node:18-slim

# Set the working directory
WORKDIR /app

# Copy package files from your /root folder (where you moved them)
COPY package.json ./

# Install dependencies (ignoring the audit)
RUN npm install --no-audit

# Copy the rest of your application code
COPY . .

# Build the Next.js app
RUN npm run build

# Start the application
CMD ["npm", "start"]
