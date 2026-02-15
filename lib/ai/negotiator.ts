// lib/ai/negotiator.ts

export const generateNegotiationScript = (diagnosis: any, priceData: any) => {
  const { primary_issue, estimated_cost } = diagnosis;
  const { asking_price } = priceData;
  
  const repairTotal = estimated_cost.total.avg;
  const targetPrice = asking_price - repairTotal;

  return {
    opening: "I’ve completed a digital diagnostic scan of the vehicle, and I’m very interested.",
    the_fact: `The scan identified a specific issue: ${primary_issue}.`,
    the_math: `Market repair data for this specific model indicates an upcoming cost of approximately $${repairTotal}.`,
    the_ask: `Because of this, I’m looking for a price adjustment to $${targetPrice}. If we can agree on that, I’m ready to move forward.`,
    backup_lever: "If we can't adjust the price, would you be willing to include a comprehensive powertrain warranty to cover this specific risk?"
  };
};
