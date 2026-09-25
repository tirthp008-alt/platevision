import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PlateSight Live | Phone Number Plate Scanner',
  description: 'Live number plate scanning from a phone camera or CCTV image.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
