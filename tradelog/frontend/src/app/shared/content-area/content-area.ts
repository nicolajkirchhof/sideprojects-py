import { Component, input } from '@angular/core';

import { AngularSplitModule } from 'angular-split';

@Component({
  selector: 'app-content-area',
  standalone: true,
  imports: [AngularSplitModule],
  templateUrl: './content-area.html',
  host: {class: 'flex flex-col flex-1'},
})
export class ContentArea {
  showRightSidebar = input(false);
  sidebarWidth = input('320px');
  title = input<string>();
}
