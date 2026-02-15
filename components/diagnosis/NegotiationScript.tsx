// components/diagnosis/NegotiationScript.tsx

import { Card, Text, Bold, Button } from "@tremor/react";

export default function NegotiationScript({ script }) {
  const copyToClipboard = () => navigator.clipboard.writeText(Object.values(script).join(' '));

  return (
    <Card className="mt-6 border-l-4 border-blue-500 bg-blue-50">
      <div className="flex justify-between items-center mb-4">
        <Bold className="text-blue-700">Lot-Side Negotiation Script</Bold>
        <Button size="xs" variant="secondary" onClick={copyToClipboard}>Copy Script</Button>
      </div>
      
      <div className="space-y-3 text-slate-700 italic">
        <Text>"{script.opening}"</Text>
        <Text>"{script.the_fact}"</Text>
        <Text>"{script.the_math}"</Text>
        <Text className="font-semibold text-slate-900">"{script.the_ask}"</Text>
      </div>

      <div className="mt-4 p-3 bg-white rounded border border-blue-100">
        <Text className="text-xs uppercase tracking-widest text-blue-500 font-bold">Pro Tip</Text>
        <Text className="text-sm">If they say no, use the backup: <Bold>{script.backup_lever}</Bold></Text>
      </div>
    </Card>
  );
}
