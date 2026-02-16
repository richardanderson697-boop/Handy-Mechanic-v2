import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '../../../lib/database';
import { getUserFromHeaders } from '../../../lib/auth';
import {
  decodeVIN,
  getVINRecalls,
  getVehicleComplaints,
  getVehicleHistory,
  estimateVehicleValue,
  isValidVIN,
} from '../../../lib/vin-decoder';

export async function POST(request: NextRequest) {
  try {
    const tokenPayload = getUserFromHeaders(request.headers);
    if (!tokenPayload) {
      return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 });
    }

    const { data: user } = await supabaseAdmin
      .from('users')
      .select('*')
      .eq('id', tokenPayload.userId)
      .single();

    if (!user) {
      return NextResponse.json({ success: false, error: 'User not found' }, { status: 404 });
    }

    const body = await request.json();
    const { vin, mileage, condition } = body;

    if (!vin || !isValidVIN(vin)) {
      return NextResponse.json({ success: false, error: 'Valid VIN is required' }, { status: 400 });
    }

    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const { data: cachedHistory } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .select('*')
      .eq('vin', vin)
      .gte('created_at', thirtyDaysAgo.toISOString())
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (cachedHistory) {
      return NextResponse.json({ success: true, cached: true, history: cachedHistory }, { status: 200 });
    }

    const [vehicleInfo, recalls, vehicleHistory] = await Promise.all([
      decodeVIN(vin),
      getVINRecalls(vin),
      getVehicleHistory(vin),
    ]);

    if (!vehicleInfo) {
      return NextResponse.json({ success: false, error: 'Failed to decode VIN' }, { status: 404 });
    }

    const complaints = await getVehicleComplaints(vehicleInfo.make, vehicleInfo.model, vehicleInfo.year);
    const estimatedValue = await estimateVehicleValue(vehicleInfo.make, vehicleInfo.model, vehicleInfo.year, mileage, condition);

    const { data: savedHistory } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .insert([{
        user_id: user.id,
        vin,
        nhtsa_data: vehicleInfo,
        recalls: recalls.length > 0 ? recalls : null,
        complaints: complaints.length > 0 ? complaints : null,
        vehicle_history: vehicleHistory,
        estimated_value: estimatedValue,
        credits_used: 0,
      }])
      .select()
      .single();

    return NextResponse.json({ success: true, cached: false, history: savedHistory }, { status: 200 });

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json({ success: false, error: 'Internal Server Error' }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  try {
    const tokenPayload = getUserFromHeaders(request.headers);
    if (!tokenPayload) {
      return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 });
    }
    const { data: histories } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .select('*')
      .eq('user_id', tokenPayload.userId)
      .order('created_at', { ascending: false });

    return NextResponse.json({ success: true, histories }, { status: 200 });
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Failed to fetch' }, { status: 500 });
  }
}
