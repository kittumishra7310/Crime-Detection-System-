import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: Request) {
  try {
    const authHeader = request.headers.get('authorization');

    const response = await fetch(`${API_BASE_URL}/api/cameras`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { 'Authorization': authHeader }),
        'host': new URL(API_BASE_URL).host,
      },
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || 'Failed to fetch cameras' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Proxy error in /api/proxy/cameras:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}
