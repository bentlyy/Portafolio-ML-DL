import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('renders ML Portfolio navigation', () => {
    render(<App />);
    expect(screen.getByText('ML Portfolio')).toBeInTheDocument();
  });
});
