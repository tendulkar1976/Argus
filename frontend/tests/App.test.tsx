import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../src/App';

describe('Argus Frontend Smoke Test', () => {
  it('renders Argus brand header and Phase 1 title', () => {
    render(<App />);
    expect(screen.getByText('Argus')).toBeInTheDocument();
    expect(screen.getByText('Phase 1')).toBeInTheDocument();
    expect(
      screen.getByText('Cost-Efficient LLM Inference Architecture')
    ).toBeInTheDocument();
  });

  it('renders verify backend health button', () => {
    render(<App />);
    expect(
      screen.getByRole('button', { name: /verify backend health/i })
    ).toBeInTheDocument();
  });
});
