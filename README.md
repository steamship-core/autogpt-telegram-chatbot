# Tutorial: Deploy your own AutoGPT Telegram bot

💛 Credits to [yoheinakajima](https://twitter.com/yoheinakajima) for bringing [babyagi](https://github.com/yoheinakajima/babyagi) to life.

This project contains the necessary scaffolding to deploy your own AutoGPT/BabyAGI via LangChain agents with memory.

These 3 steps should get you online. If not, shoot me a message on [Discord](https://steamship.com/discord). Happy to help you out. 


Let's go: 

<br>

<details open><summary>The no-coder way 😎</summary>

  1. Go to https://www.steamship.com/build/auto-gpt 

  2. Click **launch** 

  3. Copy-paste your bot token and click create 🚀
</details>


<details><summary>The coder way 💻</summary>

1. **Step 1:** Edit the AutoGPT agent in `src/babyagi.py`. You can also use the vanilla babyagi in the file. 


3. **Step 2:** Pip install the latest `steamship_langchain`: `pip install --upgrade steamship_langchain`


4. **Step 3:** Run `python deploy.py`. The script will ask you to copy-paste your bot token. Learn how to get a Telegram bot token [here](docs/register-telegram-bot.md).
</details>

