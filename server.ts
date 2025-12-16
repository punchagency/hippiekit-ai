import dotenv from 'dotenv';
dotenv.config();

import express, { Request, Response } from 'express';
import cors from 'cors';
import authRoutes from './routes/auth.js';
import favoriteRoutes from './routes/favorites.js';
import wordpressRoutes from './routes/wordpress.js';
import connectDB from './config/db.js';

const PORT = process.env.PORT || 8000;
const app = express();

// === Allowed origins for CORS ===
const allowedOrigins = [
  'http://localhost:5173', // Vite dev server
  'capacitor://localhost', // Capacitor WebView
  'https://localhost', // Capacitor HTTPS
  process.env.APP_URL, // Production URL (Heroku or custom domain)
].filter(Boolean);

// === Environment validation ===
const requiredEnv = ['JWT_SECRET', 'ATLAS_URL'];
requiredEnv.forEach((key) => {
  if (!process.env[key]) {
    console.error(`❌ Missing required environment variable: ${key}`);
  }
});
if (!process.env.JWT_SECRET || !process.env.ATLAS_URL) {
  console.warn(
    '⚠️ Server starting with missing env vars may cause runtime failures (token generation or DB connection).'
  );
}

// === Middleware ===
// CORS configuration - bearer token auth doesn't need credentials
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps, Postman, curl)
      if (!origin) return callback(null, true);

      // Allow whitelisted origins
      if (allowedOrigins.includes(origin)) {
        return callback(null, true);
      }

      // Log rejected origin but still allow it (for debugging)
      console.log('⚠️ Origin not in whitelist but allowing:', origin);
      return callback(null, true);
    },
    credentials: false, // Disabled - using bearer tokens instead of cookies
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'X-Requested-With',
      'Accept',
    ],
    exposedHeaders: [
      'set-auth-token', // Bearer token header
      'Authorization',
    ],
  })
);

// Parse JSON bodies
app.use(express.json());

// Connect to MongoDB
connectDB();

// === Custom Authentication Routes ===
app.use('/api/auth', authRoutes);
// === Favorites Routes ===
app.use('/api/favorites', favoriteRoutes);
// === WordPress Proxy Routes ===
app.use('/api/wordpress', wordpressRoutes);

// Health check
app.get('/', (_req: Request, res: Response) => {
  res.json({ message: 'API is running...' });
});

// Start the Express server
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
