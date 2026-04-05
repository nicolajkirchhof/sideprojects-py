import { Component, inject, signal } from '@angular/core';

import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { AccountsService, Account } from '../../accounts/accounts.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-account-switcher',
  standalone: true,
  imports: [MatSelectModule, MatFormFieldModule],
  template: `
    <mat-form-field appearance="outline" subscriptSizing="dynamic" class="account-switcher">
      <mat-select [value]="service.selectedAccountId()" (selectionChange)="onChange($event.value)" placeholder="Select account">
        @for (a of accounts(); track a.id) {
          <mat-option [value]="a.id">{{ a.name }}</mat-option>
        }
      </mat-select>
    </mat-form-field>
  `,
  styles: [`
    .account-switcher {
      width: 140px;
    }
    :host ::ng-deep .mat-mdc-form-field-subscript-wrapper { display: none; }
    :host ::ng-deep .mdc-text-field { height: 32px; }
    :host ::ng-deep .mat-mdc-select-value { font-size: 13px; }
    :host ::ng-deep .mdc-notched-outline__leading,
    :host ::ng-deep .mdc-notched-outline__trailing,
    :host ::ng-deep .mdc-notched-outline__notch {
      border-color: rgba(255,255,255,0.3) !important;
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
