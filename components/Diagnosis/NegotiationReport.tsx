// components/Diagnosis/NegotiationReport.tsx
import { Card, Title, Text, List, ListItem, Bold, Divider } from "@tremor/react";

export default function NegotiationReport({ diagnosis, marketValue }) {
  const repairTotal = diagnosis.estimated_cost.total.min;
  const suggestedOffer = marketValue - repairTotal;

  return (
    <Card className="max-w-md mx-auto">
      <Title className="text-emerald-600">Your Negotiation Power</Title>
      <Text className="mt-2">Based on our scan, here is your bargaining leverage:</Text>
      
      <div className="bg-slate-50 p-4 rounded-lg my-4">
        <Text>Asking Price: <Bold>${marketValue}</Bold></Text>
        <Text className="text-rose-500">Identified Repairs: <Bold>-${repairTotal}</Bold></Text>
        <Divider />
        <Text>Suggested Counter-Offer: <Bold className="text-xl">${suggestedOffer}</Bold></Text>
      </div>

      <Title className="text-sm mt-6">What to say to the dealer:</Title>
      <div className="bg-blue-50 p-4 rounded-lg italic text-slate-700">
        "I’m interested in the car, but my independent inspection scan flagged <Bold>{diagnosis.primary_issue}</Bold>. 
        The estimated cost to address this is <Bold>${repairTotal}</Bold>. If you can adjust the price to <Bold>${suggestedOffer}</Bold>, I'm ready to move forward today."
      </div>
    </Card>
  );
}
