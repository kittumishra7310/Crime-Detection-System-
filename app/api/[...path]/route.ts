import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// This function handles all HTTP methods
async function handler(req: NextRequest) {
  const path = req.nextUrl.pathname;
  const backendUrl = `${API_BASE_URL}${path}`;

  try {
    const response = await fetch(backendUrl, {
      method: req.method,
      headers: {
        ...req.headers,
        host: new URL(API_BASE_URL).host, // Set the host header to the backend's host
      },
      body: req.body,
      cache: 'no-store',
    });

    return response;

  } catch (error: any) {
    console.error('API Proxy error:', error);
    return new NextResponse(`Proxy error: ${error.message}`, { status: 500 });
  }
}

export { handler as GET, handler as POST, handler as PUT, handler as DELETE, handler as PATCH };
