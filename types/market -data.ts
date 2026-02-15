// types/market-data.ts

export interface MarketCostEstimate {
  symptom_detected: string; // e.g., "Worn Brake Pads"
  market_rate: {
    parts_low: number;
    parts_high: number;
    labor_low: number;
    labor_high: number;
    avg_total: number;
  };
  negotiation_leverage: string; // "High - This is a safety item"
  diy_savings: number;
  urgency_multiplier: number; // For immediate repairs
}

// Example use-case: The AI hears a squeal and sees thin pads.
// It returns: "Estimated $450 in upcoming maintenance."
