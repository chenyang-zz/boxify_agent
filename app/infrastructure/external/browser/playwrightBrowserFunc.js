// 获取当前页面可见内容的所有元素

const getVisibileContent = () => {
  const visibleElements = [];
  const viewportHeight = window.innerHeight;
  const viewportWidth = window.innerWidth;

  const elements = document.querySelectorAll("body *");

  for (const element of elements) {
    // 获取元素的位置和大小
    const rect = element.getBoundingClientRect();

    // 过滤掉不可见的元素
    if (rect.height === 0 || rect.width === 0) continue;

    // 判断元素是否在视口中
    if (
      rect.bottom < 0 ||
      rect.top > viewportHeight ||
      rect.left > viewportWidth ||
      rect.right < 0
    )
      continue;

    // 过滤掉不可见的元素
    const style = window.getComputedStyle(element);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.opacity === "0"
    )
      continue;

    // 添加有意义的节点
    if (
      element.innerText ||
      element.tagName === "IMG" ||
      element.tagName === "INPUT" ||
      element.tagName === "BUTTON" ||
      element.tagName === "TEXTAREA"
    ) {
      visibleElements.push(element.outerHTML);
    }
  }

  return "<div>" + visibleElements.join(" ") + "</div>";
};

const getInteractiveElements = () => {
  const interactiveElements = [];
  const viewportHeight = window.innerHeight;
  const viewportWidht = window.innerWidth;

  // 遍历所有可交互的元素
  const elements = document.querySelectorAll(
    'button, a, input, textarea, select, [role="button"], [tabindex]:not([tabindex="-1"])',
  );

  // 定义变量用于生成连续的唯一索引
  let validElementIndex = 0;

  for (const element of elements) {
    const rect = element.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) continue;

    if (
      rect.bottom < 0 ||
      rect.top > viewportHeight ||
      rect.left > viewportWidht ||
      rect.right < 0
    )
      continue;

    const style = window.getComputedStyle(element);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.opacity === "0"
    )
      continue;

    let tagName = element.tagName.toLowerCase();
    let text = "";

    if (element.value && ["input", "textarea", "select"].includes(tagName)) {
      text = element.value;

      // 如果是 input 元素，尝试获取标签文本
      if (tagName === "input") {
        let labelText = "";
        if (element.id) {
          const label = document.querySelector(`label[for="${element.id}"]`);
          if (label) {
            labelText = label.innerText.trim();
          }
        }

        // 如果标签文本为空，尝试从父级 label 中获取
        if (!labelText) {
          const parentLabel = element.closest("label");
          if (parentLabel) {
            labelText = parentLabel.innerText
              .trim()
              .replace(element.value, "")
              .trim();
          }
        }

        // 拼接label
        if (labelText) {
          text = `[Label: ${labelText}] ${text}`;
        }

        // 如果有 placeholder，拼接 placeholder
        if (element.placeholder) {
          text = `${text} [Placeholder: ${element.placeholder}]`;
        }
      } else if (element.innerText) {
        // 普通元素则提取内部文本并剔除多余空格 (如 <button>提交</button>)
        text = element.innerText.trim().replace(/\\s+/g, " ");
      }
    } else if (element.alt) {
      // 图片按钮，取 alt 属性
      text = element.alt;
    } else if (element.title) {
      // 取 title 属性
      text = element.title;
    } else if (element.placeholder) {
      // 取 placeholder 属性
      text = element.placeholder;
    } else if (element.type) {
      // 兜底逻辑将元素的类型作为文本描述
      text = `[${element.type}]`;

      // 针对没有值的 Input，再次尝试获取 Label 和 Placeholder (逻辑同上)
      if (tagName === "input") {
        let labelText = "";
        if (element.id) {
          const label = document.querySelector(`label[for="${element.id}"]`);
          if (label) {
            labelText = label.innerText.trim();
          }
        }

        if (!labelText) {
          const parentLabel = element.closest("label");
          if (parentLabel) {
            labelText = parentLabel.innerText.trim();
          }
        }

        if (labelText) {
          text = `[Label: ${labelText}] ${text}`;
        }

        if (element.placeholder) {
          text = `${text} [Placeholder: ${element.placeholder}]`;
        }
      }
    } else {
      // 都不满足，则设置为No text
      text = "[No text]";
    }

    // 截断文本，避免过长
    if (text.length > 100) {
      text = text.substring(0, 97) + "...";
    }

    // 添加属性
    element.setAttribute(
      "data-boxify-id",
      `boxify-element-${validElementIndex}`,
    );

    // 生成选择器
    const selector = `[data-boxify-id="boxify-element-${validElementIndex}"]`;

    // 将元素信息添加到交互元素列表
    interactiveElements.push({
      index: validElementIndex,
      tag: tagName,
      text: text,
      selector: selector,
    });

    validElementIndex++;
  }

  return interactiveElements;
};

const injectConsoleLogs = () => {
  window.console.logs = [];

  const originLog = console.log;

  console.log = (...args) => {
    window.console.logs.push(args.join(" "));
    originLog.apply(console, args);
  };
};
