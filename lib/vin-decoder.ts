import type { VINDecodeResult, NHTSARecall } from '../types';

/**
 * Decode VIN using NHTSA API (free, no API key required)
 */
export async function decodeVIN(vin: string): Promise<VINDecodeResult | null> {
  if (!vin || vin.length !== 17) {
    throw new Error('Invalid VIN format');
  }

  try {
    const response = await fetch(
      `https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/${vin}?format=json`
    );

    if (!response.ok) {
      throw new Error('VIN decode failed');
    }

    const data = await response.json();
    const result = data.Results?.[0];

    if (!result || result.ErrorCode !== '0') {
      return null;
    }

    return {
      make: result.Make || '',
      model: result.Model || '',
      year: parseInt(result.ModelYear) || 0,
      trim: result.Trim || undefined,
      engine: result.EngineModel || result.EngineCylinders ? 
        `${result.EngineCylinders || ''} ${result.EngineModel || ''}`.trim() : undefined,
      transmission: result.TransmissionStyle || undefined,
      drivetrain: result.DriveType || undefined,
      bodyStyle: result.BodyClass || undefined,
      manufacturerName: result.Manufacturer || undefined,
      plantCountry: result.PlantCountry || undefined,
    };
  } catch (error) {
    console.error('[v0] VIN decode error:', error);
    return null;
  }
}

/**
 * Get recalls for a VIN from NHTSA (free API)
 */
export async function getVINRecalls(vin: string): Promise<NHTSARecall[]> {
  if (!vin || vin.length !== 17) {
    return [];
  }

  try {
    const response = await fetch(
      `https://api.nhtsa.gov/recalls/recallsByVehicle?vin=${vin}`
    );

    if (!response.ok) {
      return [];
    }

    const data = await response.json();
    const recalls = data.results || [];

    return recalls.map((recall: any) => ({
      recallNumber: recall.NHTSACampaignNumber,
      component: recall.Component,
      summary: recall.Summary,
      consequence: recall.Consequence,
      remedy: recall.Remedy,
      recallDate: recall.ReportReceivedDate,
    }));
  } catch (error) {
    console.error('[v0] Recall fetch error:', error);
    return [];
  }
}

/**
 * Get vehicle complaints from NHTSA
 */
export async function getVehicleComplaints(
  make: string,
  model: string,
  year: number
): Promise<any[]> {
  try {
    const response = await fetch(
      `https://api.nhtsa.gov/complaints/complaintsByVehicle?make=${make}&model=${model}&modelYear=${year}`
    );

    if (!response.ok) {
      return [];
    }

    const data = await response.json();
    return data.results?.slice(0, 20) || [];
  } catch (error) {
    console.error('[v0] Complaints fetch error:', error);
    return [];
  }
}

/**
 * Get vehicle history report (requires paid API - Carfax/AutoCheck equivalent)
 * This is a placeholder for integration with vehicle history APIs
 */
export async function getVehicleHistory(vin: string): Promise<any> {
  // Check for API keys
  const carfaxKey = process.env.CARFAX_API_KEY;
  const nmvtisKey = process.env.NMVTIS_API_KEY;
  const vehicleHistoryKey = process.env.VEHICLE_HISTORY_API_KEY;

  if (!carfaxKey && !nmvtisKey && !vehicleHistoryKey) {
    console.warn('[v0] No vehicle history API keys configured');
    return null;
  }

  try {
    // Example integration with NMVTIS (National Motor Vehicle Title Information System)
    if (nmvtisKey) {
      const response = await fetch(
        `https://www.vehiclehistory.gov/nmvtis/api/report`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${nmvtisKey}`,
          },
          body: JSON.stringify({ vin }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        return {
          provider: 'NMVTIS',
          vin,
          titleInfo: data.titleInfo,
          brandHistory: data.brandHistory,
          junkSalvageHistory: data.junkSalvage,
          odometer: data.odometer,
          lastReportedMileage: data.lastReportedMileage,
        };
      }
    }

    // Placeholder for Carfax integration
    if (carfaxKey) {
      // Carfax API integration would go here
      // Note: Carfax API requires business partnership
      console.log('[v0] Carfax API not yet implemented');
    }

    return null;
  } catch (error) {
    console.error('[v0] Vehicle history error:', error);
    return null;
  }
}

/**
 * Estimate vehicle value using market data
 * In production, integrate with KBB, Edmunds, or market data API
 */
export async function estimateVehicleValue(
  make: string,
  model: string,
  year: number,
  mileage?: number,
  condition?: 'excellent' | 'good' | 'fair' | 'poor'
): Promise<any> {
  try {
    // This is a simplified calculation
    // In production, use KBB API, Edmunds API, or similar
    
    // Base value calculation (very simplified)
    const currentYear = new Date().getFullYear();
    const age = currentYear - year;
    const baseValue = 30000; // Starting point
    const depreciationRate = 0.15; // 15% per year
    
    let estimatedValue = baseValue * Math.pow(1 - depreciationRate, age);
    
    // Adjust for mileage
    if (mileage) {
      const expectedMileage = age * 12000;
      const mileageDiff = mileage - expectedMileage;
      const mileageAdjustment = (mileageDiff / 1000) * 50;
      estimatedValue -= mileageAdjustment;
    }
    
    // Adjust for condition
    const conditionMultipliers = {
      excellent: 1.1,
      good: 1.0,
      fair: 0.85,
      poor: 0.65,
    };
    
    if (condition) {
      estimatedValue *= conditionMultipliers[condition];
    }
    
    return {
      estimatedValue: Math.round(estimatedValue),
      range: {
        min: Math.round(estimatedValue * 0.9),
        max: Math.round(estimatedValue * 1.1),
      },
      currency: 'USD',
      factors: {
        age,
        mileage,
        condition,
      },
      disclaimer: 'Estimate only. Actual value may vary based on location, condition, and market.',
    };
  } catch (error) {
    console.error('[v0] Value estimation error:', error);
    return null;
  }
}

/**
 * Validate VIN format
 */
export function isValidVIN(vin: string): boolean {
  if (!vin || vin.length !== 17) {
    return false;
  }

  // VIN should not contain I, O, or Q
  if (/[IOQ]/.test(vin)) {
    return false;
  }

  // VIN should only contain alphanumeric characters
  if (!/^[A-HJ-NPR-Z0-9]{17}$/.test(vin)) {
    return false;
  }

  return true;
}
