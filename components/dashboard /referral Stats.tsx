// components/Dashboard/ReferralStats.tsx
import { Card, Title, AreaChart, Metric, Text, Flex, Badge } from "@tremor/react";

const stats = [
  { name: "Warranty Revenue", value: "$4,250", change: "+12%", status: "top-performer" },
  { name: "Financing Leads", value: "128", change: "+5%", status: "steady" },
  { name: "Avg. EPC", value: "$1.45", change: "-2%", status: "monitor" },
];

export default function ReferralDashboard() {
  return (
    <div className="p-6">
      <Title>Affiliate Performance Overview</Title>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
        {stats.map((item) => (
          <Card key={item.name} decoration="top" decorationColor="blue">
            <Text>{item.name}</Text>
            <Metric>{item.value}</Metric>
            <Flex className="mt-4">
              <Badge color={item.change.startsWith('+') ? "emerald" : "rose"}>
                {item.change}
              </Badge>
              <Text className="truncate ml-2">from last month</Text>
            </Flex>
          </Card>
        ))}
      </div>
      {/* Charting conversion trends between AI Findings vs Partner Clicks */}
    </div>
  );
}
