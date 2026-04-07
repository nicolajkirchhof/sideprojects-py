import { Component, inject, signal } from '@angular/core';

import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { AccountsService, Account } from '../../accounts/accounts.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-account-switcher',
  standalone: true,
  imports: [MatSelectModule, MatFormFieldModule, MatIconModule, MatTooltipModule],
  template: `
    <div class="account-switcher" [matTooltip]="'Active account'">
      <mat-icon class="account-switcher__icon">account_circle</mat-icon>
      <mat-select
        [value]="service.selectedAccountId()"
        (selectionChange)="onChange($event.value)"
        panelClass="account-switcher-panel"
        placeholder="Select account">
        @for (a of accounts(); track a.id) {
          <mat-option [value]="a.id">{{ a.name }}</mat-option>
        }
      </mat-select>
    </div>
  `,
  styles: [`
    .account-switcher {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      height: 1.875rem; /* 30px */
      padding: 0 0.625rem;
      min-width: 10rem; /* 160px */
      border-radius: 0.25rem;
      background-color: var(--mat-sys-surface-container-high);
      border: 1px solid var(--mat-sys-outline-variant);
      transition: background-color 120ms ease, border-color 120ms ease;
    }
    .account-switcher:hover {
      background-color: var(--mat-sys-surface-container-highest);
      border-color: var(--mat-sys-outline);
    }
    .account-switcher__icon {
      font-size: 1rem;
      width: 1rem;
      height: 1rem;
      color: var(--mat-sys-on-surface-variant);
    }
    :host ::ng-deep .account-switcher .mat-mdc-select {
      font-size: 0.8125rem;
      font-weight: 500;
      color: var(--mat-sys-on-surface);
      letter-spacing: 0.01em;
    }
    :host ::ng-deep .account-switcher .mat-mdc-select-arrow {
      color: var(--mat-sys-on-surface-variant);
    }
    :host ::ng-deep .account-switcher .mat-mdc-select-placeholder {
      color: var(--mat-sys-on-surface-variant);
    }
  `],
})
export class AccountSwitcherComponent {
  protected service = inject(AccountsService);
  private router = inject(Router);

  accounts = signal<Account[]>([]);

  constructor() {
    this.service.getAll().subscribe({
      next: (accounts) => {
        this.accounts.set(accounts);
        if (this.service.selectedAccountId() === 0 && accounts.length > 0) {
          const defaultAccount = accounts.find(a => a.isDefault) ?? accounts[0];
          this.onChange(defaultAccount.id);
        }
      },
    });
  }

  onChange(id: number): void {
    this.service.selectAccount(id);
    const url = this.router.url;
    this.router.navigateByUrl('/', { skipLocationChange: true }).then(() => {
      this.router.navigateByUrl(url);
    });
  }
}
