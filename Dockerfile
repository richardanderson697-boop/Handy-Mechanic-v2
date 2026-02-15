# Use a lightweight Node image
FROM node:18-slim

# Set the working directory
WORKDIR /app

# Copy package files from the current directory
COPY package.json ./

# Force install without audit and skip devDependencies to reduce risk
RUN npm install --no-audit --fund=false

# Copy the rest of your application code
COPY . .

# Build the Next.js app
RUN npm run build

# Start the application
CMD ["npm", "start"]
