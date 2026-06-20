/**
 * App.tsx — Root application component.
 * Renders the Navbar and the Home page.
 */

import React from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';

const App: React.FC = () => {
  return (
    <>
      <Navbar />
      <Home />
    </>
  );
};

export default App;
