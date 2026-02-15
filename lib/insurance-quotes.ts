import type { InsuranceQuoteRequest, InsuranceQuoteResult } from '../types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Insurance Quote Service
 * Integrates with major insurance providers via affiliate programs
 */

interface InsuranceProvider {
  id: string;
  name: string;
  baseUrl: string;
  affiliateId?: string;
  apiKey?: string;
}

const PROVIDERS: InsuranceProvider[] = [
  {
    id: 'progressive',
    name: 'Progressive',
    baseUrl: 'https://www.progressive.com',
    affiliateId: process.env.PROGRESSIVE_AFFILIATE_ID,
  },
  {
    id: 'geico',
    name: 'GEICO',
    baseUrl: 'https://www.geico.com',
    affiliateId: process.env.GEICO_AFFILIATE_ID,
  },
  {
    id: 'state_farm',
    name: 'State Farm',
    baseUrl: 'https://www.statefarm.com',
    affiliateId: process.env.STATE_FARM_AFFILIATE_ID,
  },
  {
    id: 'allstate',
    name: 'Allstate',
    baseUrl: 'https://www.allstate.com',
    affiliateId: process.env.ALLSTATE_AFFILIATE_ID,
  },
  {
    id: 'liberty_mutual',
    name: 'Liberty Mutual',
    baseUrl: 'https://www.libertymutual.com',
  },
  {
    id: 'usaa',
    name: 'USAA',
    baseUrl: 'https://www.usaa.com',
  },
];

/**
 * Get insurance quotes from multiple providers
 * In production, this would integrate with insurance comparison APIs
 */
export async function getInsuranceQuotes(
  request: InsuranceQuoteRequest
): Promise<InsuranceQuoteResult[]> {
  const quotes: InsuranceQuoteResult[] = [];
  
  // Generate unique tracking code for affiliate links
  const trackingCode = uuidv4().substring(0, 8);

  for (const provider of PROVIDERS) {
    const quote = await generateQuoteForProvider(provider, request, trackingCode);
    if (quote) {
      quotes.push(quote);
    }
  }

  // Sort by monthly premium (lowest first)
  return quotes.sort((a, b) => a.monthlyPremium - b.monthlyPremium);
}

/**
 * Generate quote for a specific provider
 * In production, this would call actual insurance APIs
 * For now, we generate estimated quotes with affiliate links
 */
async function generateQuoteForProvider(
  provider: InsuranceProvider,
  request: InsuranceQuoteRequest,
  trackingCode: string
): Promise<InsuranceQuoteResult | null> {
  try {
    // Generate estimated premium based on vehicle and driver info
    const basePremium = calculateBasePremium(request);
    
    // Add provider-specific adjustments
    const providerMultiplier = getProviderMultiplier(provider.id);
    const monthlyPremium = Math.round(basePremium * providerMultiplier);

    // Build affiliate URL
    const affiliateUrl = buildAffiliateUrl(provider, request, trackingCode);
    const affiliateCode = `${provider.id}_${trackingCode}`;

    // Determine coverage details
    const coverageLevel = getCoverageLevel(request.coverageType);
    const deductible = getDeductible(request.coverageType);

    // Provider-specific features
    const features = getProviderFeatures(provider.id);

    // Provider rating (mock data - in production, use real ratings)
    const rating = getProviderRating(provider.id);

    return {
      provider: provider.name,
      monthlyPremium,
      coverageLevel,
      deductible,
      affiliateUrl,
      affiliateCode,
      features,
      rating,
    };
  } catch (error) {
    console.error(`[v0] Error generating quote for ${provider.name}:`, error);
    return null;
  }
}

/**
 * Calculate base premium based on vehicle and driver factors
 */
function calculateBasePremium(request: InsuranceQuoteRequest): number {
  let premium = 100; // Base monthly premium

  // Age adjustment
  if (request.age < 25) {
    premium *= 1.5; // Young drivers pay more
  } else if (request.age > 65) {
    premium *= 1.2;
  } else {
    premium *= 1.0; // Optimal age range
  }

  // Driving history adjustment
  const drivingMultipliers = {
    clean: 1.0,
    minor_violations: 1.3,
    major_violations: 1.8,
  };
  premium *= drivingMultipliers[request.drivingHistory];

  // Vehicle age adjustment (older vehicles generally cost less to insure)
  const vehicleAge = new Date().getFullYear() - request.vehicle.year;
  if (vehicleAge > 10) {
    premium *= 0.8;
  } else if (vehicleAge > 5) {
    premium *= 0.9;
  }

  // Coverage type adjustment
  const coverageMultipliers = {
    liability: 1.0,
    collision: 1.5,
    comprehensive: 2.0,
  };
  premium *= coverageMultipliers[request.coverageType];

  // Add some randomness for provider variation
  premium *= (0.9 + Math.random() * 0.3);

  return Math.round(premium);
}

/**
 * Get provider-specific pricing multiplier
 */
function getProviderMultiplier(providerId: string): number {
  const multipliers: Record<string, number> = {
    progressive: 0.95,
    geico: 0.90, // Often cheapest
    state_farm: 1.05,
    allstate: 1.10,
    liberty_mutual: 1.00,
    usaa: 0.85, // Best for military, but restricted
  };

  return multipliers[providerId] || 1.0;
}

/**
 * Get coverage level description
 */
function getCoverageLevel(coverageType: string): string {
  const levels: Record<string, string> = {
    liability: 'Liability Only',
    collision: 'Liability + Collision',
    comprehensive: 'Full Coverage (Comprehensive)',
  };

  return levels[coverageType] || 'Standard Coverage';
}

/**
 * Get deductible amount
 */
function getDeductible(coverageType: string): number {
  const deductibles: Record<string, number> = {
    liability: 0,
    collision: 500,
    comprehensive: 500,
  };

  return deductibles[coverageType] || 500;
}

/**
 * Get provider-specific features
 */
function getProviderFeatures(providerId: string): string[] {
  const allFeatures: Record<string, string[]> = {
    progressive: [
      'Name Your Price® Tool',
      'Snapshot® usage-based discount',
      '24/7 claims service',
      'Roadside assistance available',
    ],
    geico: [
      '15 minutes could save you 15% or more',
      'Mobile app with ID cards',
      'Military & federal employee discounts',
      'Emergency roadside service',
    ],
    state_farm: [
      'Local agent support',
      'Drive Safe & Save™ program',
      'Good student discounts',
      'Multi-policy discounts',
    ],
    allstate: [
      'Drivewise® rewards program',
      'Accident forgiveness',
      'New car replacement',
      'Safe driving bonus checks',
    ],
    liberty_mutual: [
      'Customize your coverage',
      'RightTrack® usage-based savings',
      'Better car replacement',
      '24/7 support',
    ],
    usaa: [
      'Military members only',
      'Exceptional customer service',
      'Accident forgiveness',
      'Multi-vehicle discounts',
    ],
  };

  return allFeatures[providerId] || ['Competitive rates', '24/7 support'];
}

/**
 * Get provider rating (out of 5)
 */
function getProviderRating(providerId: string): number {
  const ratings: Record<string, number> = {
    progressive: 4.2,
    geico: 4.5,
    state_farm: 4.7,
    allstate: 4.3,
    liberty_mutual: 4.0,
    usaa: 4.9,
  };

  return ratings[providerId] || 4.0;
}

/**
 * Build affiliate tracking URL
 */
function buildAffiliateUrl(
  provider: InsuranceProvider,
  request: InsuranceQuoteRequest,
  trackingCode: string
): string {
  const params = new URLSearchParams({
    zip: request.zipCode,
    year: request.vehicle.year.toString(),
    make: request.vehicle.make,
    model: request.vehicle.model,
    tracking: trackingCode,
  });

  // Add affiliate ID if available
  if (provider.affiliateId) {
    params.set('affiliate_id', provider.affiliateId);
  }

  // Provider-specific URL structures
  const urlPatterns: Record<string, string> = {
    progressive: '/quote/auto',
    geico: '/getaquote',
    state_farm: '/insurance/auto',
    allstate: '/quote/auto-insurance',
    liberty_mutual: '/auto-insurance-quote',
    usaa: '/insurance/auto-insurance-quote',
  };

  const path = urlPatterns[provider.id] || '/quote';
  
  return `${provider.baseUrl}${path}?${params.toString()}`;
}

/**
 * Track affiliate click
 */
export async function trackAffiliateClick(
  quoteId: string,
  provider: string,
  affiliateCode: string
): Promise<void> {
  console.log('[v0] Tracking affiliate click:', { quoteId, provider, affiliateCode });
  
  // In production, this would:
  // 1. Log to analytics
  // 2. Track in affiliate network
  // 3. Update database for commission tracking
}

/**
 * Validate zip code format
 */
export function isValidZipCode(zip: string): boolean {
  // US zip code: 5 digits or 5+4 format
  return /^\d{5}(-\d{4})?$/.test(zip);
}
