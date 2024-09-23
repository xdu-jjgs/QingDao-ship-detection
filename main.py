import asyncio
import logging
import sys

from run import main

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error('发生错误', exc_info=True)