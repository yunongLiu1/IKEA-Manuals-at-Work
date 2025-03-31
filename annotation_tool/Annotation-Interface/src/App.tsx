import React, { useState } from 'react';
import { AuthProvider } from './AuthContext';
import AuthWrapper from './AuthWrapper';

function App() {
  const [user, setUser] = useState<string | null>(null);

  return (
    <AuthProvider value={{ user, setUser }}>
      <AuthWrapper />
    </AuthProvider>
  );
}

export default App;
